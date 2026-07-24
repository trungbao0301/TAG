from setuptools import setup, find_packages
from glob import glob

package_name = "tag_state_estimation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test/"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("calib/*.txt")),
        ("share/" + package_name, ["markers.csv"]),
        #("share/" + package_name, "rviz/config.rviz"),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="timflueckiger",
    maintainer_email="timflueckiger@outlook.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "estimator = tag_state_estimation.tag_state_estimation_node:main",
            "estimator_sub = tag_state_estimation.tag_state_estimation_subimg:main",
            "select_markers = tag_state_estimation.select_markers:main",
            "ai_labeler = tag_state_estimation.ai_dataset_labeler:main",
            "ai_train = tag_state_estimation.train_ai_marble:main",
            "ai_detector = tag_state_estimation.ai_marble_detector_node:main",
        ],
    },
)
