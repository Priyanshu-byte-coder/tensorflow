#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Extract hermetic CUDA User-Mode Driver (UMD) flags
HERMETIC_CUDA_UMD_BUILD_FLAGS=""
HERMETIC_CUDA_UMD_TEST_FLAGS=""
if [[ "$TFCI_BAZEL_HERMETIC_CUDA_UMD_ENABLE" == 1 ]]; then
  HERMETIC_CUDA_UMD_BUILD_FLAGS="--config=hermetic_cuda_umd"
  HERMETIC_CUDA_UMD_TEST_FLAGS="--@local_config_cuda//cuda:override_include_cuda_libs=true --config=hermetic_cuda_umd"

  # Extract the UMD version resolved for this build directly from .bazelrc
  export HERMETIC_CUDA_UMD_VERSION=""
  TEST_CONFIG="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_wheel_test"
  CONFIG_LINE=$(grep "^test:${TEST_CONFIG} " "$TFCI_GIT_DIR/.bazelrc" || true)
  if [[ "$CONFIG_LINE" =~ HERMETIC_CUDA_UMD_VERSION=\"?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
    export HERMETIC_CUDA_UMD_VERSION="${BASH_REMATCH[1]}"
  else
    for conf in $(echo "$CONFIG_LINE" | grep -o -e '--config=[a-zA-Z0-9_-]*' | sed 's/--config=//'); do
      SUBCONFIG_LINE=$(grep "^test:${conf} " "$TFCI_GIT_DIR/.bazelrc" || true)
      if [[ "$SUBCONFIG_LINE" =~ HERMETIC_CUDA_UMD_VERSION=\"?([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        export HERMETIC_CUDA_UMD_VERSION="${BASH_REMATCH[1]}"
        break
      fi
    done
  fi

  # Install UMD compat library in the container
  echo "Installing UMD compat library inside the container..."
  tfrun env HERMETIC_CUDA_UMD_VERSION="$HERMETIC_CUDA_UMD_VERSION" bash -c '
    set -exo pipefail
    if [[ -z "$HERMETIC_CUDA_UMD_VERSION" ]]; then
      echo "Error: HERMETIC_CUDA_UMD_VERSION environment variable is not set."
      exit 1
    fi

    if [[ "$HERMETIC_CUDA_UMD_VERSION" =~ ^([0-9]+)\.([0-9]+) ]]; then
      COMPAT_VERSION="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
    else
      echo "Error: Invalid HERMETIC_CUDA_UMD_VERSION format ($HERMETIC_CUDA_UMD_VERSION)."
      exit 1
    fi

    echo "Setting up NVIDIA apt repository..."
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub || true
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/nvidia.list

    echo "Running apt-get to install cuda-compat-${COMPAT_VERSION}..."
    apt-get update -y
    
    # This code handles installing the correct NVIDIA User-Mode Driver (UMD) compatibility
    # libraries inside the Docker container to ensure they perfectly match the required
    # driver series, avoiding \`cudaErrorInsufficientDriver\` or DSO vs. Kernel version
    # mismatch errors during testing (e.g. from cuda_diagnostics.cc).

    # 1. Detect the Host Kernel Driver Version
    # Run nvidia-smi to see what exact GPU kernel driver is running on the host machine
    # executing the Docker container (e.g., 570.86.15).
    KERNEL_DRIVER_VERSION=""
    if command -v nvidia-smi &> /dev/null; then
      KERNEL_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
      echo "Host kernel driver version detected: $KERNEL_DRIVER_VERSION"
    fi

    # 2. Check for an Exact Kernel Match
    EXACT_PKG=""
    if [[ -n "$KERNEL_DRIVER_VERSION" ]]; then
      # Query the apt registry (apt-cache madison) to see if there is a cuda-compat package
      # that matches the host kernel version string. If found, we install it to perfectly
      # align the container user-space library with the host kernel, bypassing strict diagnostics checks.
      EXACT_PKG=$(apt-cache madison cuda-compat-${COMPAT_VERSION} | grep "$KERNEL_DRIVER_VERSION" | head -n1 | cut -d"|" -f2 | tr -d " " || true)
    fi

    if [[ -n "$EXACT_PKG" ]]; then
      echo "Found exact match for kernel driver $KERNEL_DRIVER_VERSION: $EXACT_PKG"
      apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}=${EXACT_PKG}"
    elif [[ "$HERMETIC_CUDA_UMD_VERSION" == "13.0.0" ]]; then
      # 3. Fallback for CUDA 13.0.0
      # If there is no exact kernel match, and the required UMD version is 13.0.0 (which was
      # built internally against the 570. driver series), explicitly search apt-cache for a 570. package.
      # This forces the 570 series installation since NVIDIA recently updated the generic
      # cuda-compat-13-0 package alias to point to the newer 580 series, which could conflict
      # with a host running the 570 kernel module.
      EXACT_PKG=$(apt-cache madison cuda-compat-${COMPAT_VERSION} | grep "570\." | head -n1 | cut -d"|" -f2 | tr -d " " || true)
      if [[ -n "$EXACT_PKG" ]]; then
        echo "Found 570 series match for cuda-compat-${COMPAT_VERSION}: $EXACT_PKG"
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}=${EXACT_PKG}"
      else
        echo "Could not find 570 series match. Falling back to default."
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}"
      fi
    elif [[ "$HERMETIC_CUDA_UMD_VERSION" == "13.0.2" ]]; then
      # 4. Fallback for CUDA 13.0.2
      # Similarly, 13.0.2 was built internally against the 580. driver series, so we explicitly
      # force the installation of a 580. package if it is available.
      EXACT_PKG=$(apt-cache madison cuda-compat-${COMPAT_VERSION} | grep "580\." | head -n1 | cut -d"|" -f2 | tr -d " " || true)
      if [[ -n "$EXACT_PKG" ]]; then
        echo "Found 580 series match for cuda-compat-${COMPAT_VERSION}: $EXACT_PKG"
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}=${EXACT_PKG}"
      else
        echo "Could not find 580 series match. Falling back to default."
        apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}"
      fi
    else
      # 5. Ultimate Fallback (Default)
      # If neither 13.0.0 nor 13.0.2 (e.g. CUDA 12 builds), install whatever the default
      # cuda-compat package resolves to in the apt repository.
      echo "Installing default cuda-compat-${COMPAT_VERSION} package."
      apt-get install -y --no-install-recommends "cuda-compat-${COMPAT_VERSION}"
    fi

    echo "Updating ld.so.conf.d to include /usr/local/cuda/compat..."
    echo "/usr/local/cuda/compat" > /etc/ld.so.conf.d/cuda-compat.conf
    ldconfig
    echo "Successfully installed and configured cuda-compat-${COMPAT_VERSION}."
  '
fi

# Record GPU count and CUDA version status
if [[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]]; then
  tfrun nvidia-smi
fi

# Update the version numbers for Nightly only
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  python_bin=python3
  # TODO(belitskiy): Add a `python3` alias/symlink to Windows Docker image.
  if [[ $(uname -s) = MSYS_NT* ]]; then
    python_bin="python"
  fi
  tfrun "$python_bin" tensorflow/tools/ci_build/update_version.py --nightly
  # replace tensorflow to tf_nightly in the wheel name
  export TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG="$(echo $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG | sed 's/tensorflow/tf_nightly/')"
  export TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES="$(echo $TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES | sed 's/tensorflow/tf_nightly/g')"
fi

# TODO(b/361369076) Remove the following block after TF NumPy 1 is dropped
# Move hermetic requirement lock files for NumPy 1 to the root
if [[ "$TFCI_WHL_NUMPY_VERSION" == 1 ]]; then
  cp ./ci/official/requirements_updater/numpy1_requirements/*.txt .
fi

tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --config=cuda_wheel //tensorflow/tools/pip_package:wheel $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG $HERMETIC_CUDA_UMD_BUILD_FLAGS --verbose_failures

tfrun "$TFCI_FIND_BIN" ./bazel-bin/tensorflow/tools/pip_package -iname "*.whl" -exec cp {} $TFCI_OUTPUT_DIR \;
tfrun mkdir -p ./dist
tfrun cp $TFCI_OUTPUT_DIR/*.whl ./dist
tfrun bash ./ci/official/utilities/rename_and_verify_wheels.sh

if [[ -n "$TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES" ]]; then
  # Re-build the wheel with the same config, but with different name(s), if any.
  # This is done after the rename_and_verify_wheel.sh run above, not to have
  # to contend with extra wheels there.
  for wheel_name in ${TFCI_BUILD_PIP_PACKAGE_ADDITIONAL_WHEEL_NAMES}; do
    echo "Building for additional WHEEL_NAME: ${wheel_name}"
    CURRENT_WHEEL_NAME_ARG="--repo_env=WHEEL_NAME=${wheel_name}"
    tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS build $TFCI_BAZEL_COMMON_ARGS --config=cuda_wheel //tensorflow/tools/pip_package:wheel $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $CURRENT_WHEEL_NAME_ARG $HERMETIC_CUDA_UMD_BUILD_FLAGS
    # Copy the wheel that was just created
    tfrun bash -c "$TFCI_FIND_BIN ./bazel-bin/tensorflow/tools/pip_package -iname "${wheel_name}*.whl" -printf '%T+ %p\n' | sort | tail -n 1 | awk '{print \$2}' | xargs -I {} cp {} $TFCI_OUTPUT_DIR"
  done
fi

if [[ "$TFCI_ARTIFACT_STAGING_GCS_ENABLE" == 1 ]]; then
  # Note: -n disables overwriting previously created files.
  # TODO(b/389744576): Remove when gsutil is made to work properly on MSYS2.
  if [[ $(uname -s) != MSYS_NT* ]]; then
    gcloud storage cp -n "$TFCI_OUTPUT_DIR"/*.whl "$TFCI_ARTIFACT_STAGING_GCS_URI"
  else
    powershell -command "gcloud storage cp -n '$TFCI_OUTPUT_DIR/*.whl' '$TFCI_ARTIFACT_STAGING_GCS_URI'"
  fi
fi

if [[ "$TFCI_WHL_BAZEL_TEST_ENABLE" == 1 ]]; then
  tfrun bazel $TFCI_BAZEL_BAZELRC_ARGS test $TFCI_BAZEL_COMMON_ARGS $TFCI_BUILD_PIP_PACKAGE_BASE_ARGS $TFCI_BUILD_PIP_PACKAGE_WHEEL_NAME_ARG $HERMETIC_CUDA_UMD_TEST_FLAGS --repo_env=TF_PYTHON_VERSION=$TFCI_PYTHON_VERSION --config="${TFCI_BAZEL_TARGET_SELECTING_CONFIG_PREFIX}_wheel_test"
fi
