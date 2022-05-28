from setuptools import Extension, setup
from Cython.Build import cythonize
import os
from pathlib import Path

library_name = "compiled"

for file in os.listdir("compiled_src"):
    if file.endswith(".so"):
        print(os.path.join("compiled_src", file))
        os.remove(os.path.join("compiled_src", file))

    if file.endswith(".c"):
        print(os.path.join("compiled_src", file))
        os.remove(os.path.join("compiled_src", file))

sourcefiles = []

for root, dirs, files in os.walk("./compiled_src"):
    for file in files:
        if file.endswith(".pyx"):
            sourcefiles.append(os.path.join(root, file))

print(f"Detected {len(sourcefiles)} files")

extensions = [Extension(library_name, sourcefiles)]

setup(
    name='Compiled Library',
    # ext_modules=cythonize(extensions),
    ext_modules=cythonize(["compiled_src/**/*.pyx"],
                          compiler_directives={"always_allow_keywords": True}),
    zip_safe=False,
)
