from setuptools import setup, find_packages

package_name = "cyberrunner_dreamer_thomas"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test/"]),
    # package_data={"cyberrunner_dreamer_thomas": ["data/*.txt"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name,
            [
                "data/path_0002_hard.pkl",
                "data/path_custom.pkl",
                "data/map.DXF",
                "data/path.DXF",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Thomas Bi",
    maintainer_email="bit@ethz.ch",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_thomas = cyberrunner_dreamer_thomas.train:main",
            "train_tcp_thomas = cyberrunner_dreamer_thomas.train_tcp:main",
            "train_tcp_dreamer4_thomas = cyberrunner_dreamer_thomas.train_tcp_dreamer4:main",
            "train_parallel_thomas = cyberrunner_dreamer_thomas.train_parallel:main",
            "test_thomas = cyberrunner_dreamer_thomas.test_motors:main",
            "eval_thomas = cyberrunner_dreamer_thomas.eval:main",
        ],
    },
)
