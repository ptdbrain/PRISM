from setuptools import find_packages, setup


setup(
    name="prism",
    version="0.1.0",
    packages=find_packages(include=["prism", "prism.*"]),
    install_requires=["torch>=2.11.0", "scipy>=1.17.0"],
    entry_points={
        "console_scripts": [
            "prism-train-meta=prism.cli.train_meta:main",
            "prism-profile=prism.cli.profile:main",
            "prism-assign=prism.cli.assign:main",
            "prism-quic=prism.cli.quic:main",
            "prism-precompute-rtn=prism.cli.precompute_rtn:main",
            "prism-run=prism.cli.run:main",
            "prism-demo=prism.cli.demo:main",
        ]
    },
)
