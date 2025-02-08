import json
import pytest
from dependency_detector import generate_required_files, DependencyDetectorError

# Helper classes to simulate the LLM response
class FakeMessage:
    def __init__(self, content):
        self.content = content

class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)

class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

class FakeCompletions:
    def __init__(self, fake_response):
        self.fake_response = fake_response

    def create(self, model, messages):
        return self.fake_response

class FakeChat:
    def __init__(self, fake_completions):
        self.completions = fake_completions

class FakeLLMClient:
    def __init__(self, fake_content):
        self.chat = FakeChat(FakeCompletions(FakeResponse(fake_content)))


def test_generate_required_files_plain(monkeypatch):
    """
    Test generate_required_files with a plain JSON response from the LLM.
    """
    fake_content = '["file1.py", "file2.py"]'
    monkeypatch.setattr("dependency_detector.OpenAI", lambda: FakeLLMClient(fake_content))
    
    task = "Implement new feature X"
    context_file = "Some context file content for feature X."
    file_list = generate_required_files(task, context_file)
    
    assert file_list == ["file1.py", "file2.py"]


def test_generate_required_files_markdown(monkeypatch):
    """
    Test generate_required_files when the LLM response is wrapped in markdown formatting.
    """
    fake_content = "```json\n[\"moduleA.py\", \"moduleB.py\"]\n```"
    monkeypatch.setattr("dependency_detector.OpenAI", lambda: FakeLLMClient(fake_content))
    
    task = "Fix bug in authentication module"
    context_file = "Relevant context details for the authentication bug."
    file_list = generate_required_files(task, context_file)
    
    assert file_list == ["moduleA.py", "moduleB.py"]


def test_generate_required_files_invalid_json(monkeypatch):
    """
    Test that DependencyDetectorError is raised when LLM returns non-JSON content.
    """
    fake_content = "This is not JSON"
    monkeypatch.setattr("dependency_detector.OpenAI", lambda: FakeLLMClient(fake_content))
    
    task = "Test invalid JSON response"
    context_file = "Some context"
    
    with pytest.raises(DependencyDetectorError):
        generate_required_files(task, context_file)


def test_generate_required_files_wrong_type(monkeypatch):
    """
    Test that DependencyDetectorError is raised when the JSON from the LLM is valid,
    but not a list.
    """
    fake_content = '{"key": "value"}'  # valid JSON but not a list
    monkeypatch.setattr("dependency_detector.OpenAI", lambda: FakeLLMClient(fake_content))
    
    task = "Test wrong JSON type"
    context_file = "Another context content"
    
    with pytest.raises(DependencyDetectorError):
        generate_required_files(task, context_file)
