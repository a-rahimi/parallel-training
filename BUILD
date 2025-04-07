load("@//tools/jupyter:rules.bzl", "jupyter_notebook")

cruise_py_library(
    "parallelsim",
    srcs=["parallelsim.py"],
    deps=[
        pip_dep("matplotlib"),
        pip_dep("numpy"),
        pip_dep("pandas"),
    ],
    import_from_root=False,
)

jupyter_notebook(
    "parallelsim_notebook",
    deps=[":parallelsim"],
)
