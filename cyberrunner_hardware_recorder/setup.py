from setuptools import find_packages, setup


package_name = "cyberrunner_hardware_recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CyberRunner Team",
    maintainer_email="trungbao@example.invalid",
    description="Subscriber-only CyberRunner hardware telemetry recorder.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "record = cyberrunner_hardware_recorder.recorder:main",
            "analyze = cyberrunner_hardware_recorder.analyze:main",
            "export_sample = cyberrunner_hardware_recorder.sample:main",
        ],
    },
)
