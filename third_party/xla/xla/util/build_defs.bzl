"""Contains embed_files build rule."""

load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

visibility(DEFAULT_LOAD_VISIBILITY)

def text_to_binary_proto(name, src, proto_name, proto_file, out, **kwargs):
    """Converts a textproto to a binary proto using protoc.

    This is a simplified, OSS-friendly version of the proto_data rule.

    Args:
        name: The name of the target.
        src: The source textproto file.
        out: The name of output file.
        proto_name: The fully qualified name of the proto message.
        proto_file: The .proto file containing the message definition.
        **kwargs: Additional arguments passed to the genrule.
    """
    native.genrule(
        name = name,
        srcs = [src, proto_file],
        outs = [out],
        cmd = "$(location @com_google_protobuf//:protoc) --encode=%s $(location %s) < $(location %s) > $@" % (proto_name, proto_file, src),
        tools = ["@com_google_protobuf//:protoc"],
        **kwargs
    )
