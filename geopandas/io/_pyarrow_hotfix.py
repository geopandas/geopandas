from packaging.version import Version

import pyarrow

_ERROR_MSG = """\
Disallowed deserialization of 'arrow.py_extension_type':
storage_type = {storage_type}
serialized = {serialized}
pickle disassembly:\n{pickle_disassembly}

Reading of untrusted Parquet or Feather files with a PyExtensionType column
allows arbitrary code execution.
If you trust this file, you can enable reading the extension type by one of:

- upgrading to pyarrow >= 14.0.1, and call `pa.PyExtensionType.set_auto_load(True)`
- install pyarrow-hotfix (`pip install pyarrow-hotfix`) and disable it by running
  `import pyarrow_hotfix; pyarrow_hotfix.uninstall()`

We strongly recommend updating your Parquet/Feather files to use extension types
derived from `pyarrow.ExtensionType` instead, and register this type explicitly.
See https://arrow.apache.org/docs/dev/python/extending_types.html#defining-extension-types-user-defined-types
for more details.
"""


def patch_pyarrow():
    # starting from pyarrow 14.0.1, it has its own mechanism
    if Version(pyarrow.__version__) >= Version("14.0.1"):
        return

    # if the user has pyarrow_hotfix (https://github.com/pitrou/pyarrow-hotfix)
    # installed, use this instead (which also ensures it works if they had
    # called `pyarrow_hotfix.uninstall()`)
    try:
        import pyarrow_hotfix  # noqa: F401
    except ImportError:
        pass
    else:
        return

    # if the hotfix is already installed and enabled
    if getattr(pyarrow, "_hotfix_installed", False):
        return

    class ForbiddenExtensionType(pyarrow.ExtensionType):
        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            import io
            import pickletools

            out = io.StringIO()
            pickletools.dis(serialized, out)
            raise RuntimeError(
                _ERROR_MSG.format(
                    storage_type=storage_type,
                    serialized=serialized,
                    pickle_disassembly=out.getvalue(),
                )
            )

    pyarrow.unregister_extension_type("arrow.py_extension_type")
    pyarrow.register_extension_type(
        ForbiddenExtensionType(pyarrow.null(), "arrow.py_extension_type")
    )

    pyarrow._hotfix_installed = True


patch_pyarrow()
