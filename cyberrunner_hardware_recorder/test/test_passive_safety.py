import ast
from pathlib import Path


RECORDER = (
    Path(__file__).parents[1]
    / "cyberrunner_hardware_recorder"
    / "recorder.py"
)


def test_recorder_only_creates_subscriptions():
    tree = ast.parse(RECORDER.read_text())
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {
        "create_publisher",
        "create_service",
        "create_client",
        "create_action_client",
        "publish",
        "call",
        "call_async",
        "send_goal",
        "send_goal_async",
    }
    assert "create_subscription" in called_attributes
    assert "destroy_publisher" in called_attributes
    assert not (called_attributes & forbidden)


def test_recorder_has_no_ros_service_or_action_imports():
    tree = ast.parse(RECORDER.read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("srv" in name.split(".") for name in imported)
    assert not any("action" in name.split(".") for name in imported)
