from setuptools import Extension, setup
from Cython.Build import cythonize
import os
from pathlib import Path

library_name = "compiled"

source_folder: str = "compiled_src"

# Path(source_folder, "compiled").mkdir(exist_ok=True)

for subdir, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".so"):
            print(os.path.join(subdir, file))
            os.remove(os.path.join(subdir, file))

        if file.endswith(".c"):
            print(os.path.join(subdir, file))
            os.remove(os.path.join(subdir, file))

ext_modules = [
    Extension('mlflow_helper', ["compiled_src/mlflow_helper/*.pyx"]),
    Extension('data_management', ["compiled_src/data_management/*.pyx"])
]

setup(
    name='Compiled Library',
    # ext_modules=cythonize(extensions),
    ext_modules=cythonize(["compiled_src/**/*.pyx"],
                          compiler_directives={"always_allow_keywords": True, "language_level": 3,
                                               "emit_code_comments": True}),
    # ext_modules=cythonize(ext_modules, compiler_directives={"always_allow_keywords": True, "language_level": 3,
    #                                                        "emit_code_comments": True}),
    zip_safe=False,
)
