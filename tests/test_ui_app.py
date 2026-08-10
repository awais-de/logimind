"""Headless tests for ui/app.py using Streamlit's AppTest."""

from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def test_app_shows_api_connected_when_health_check_succeeds() -> None:
    health_response = Mock(status_code=200)

    with patch("httpx.get", return_value=health_response):
        at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert [s.value for s in at.sidebar.success] == ["API connected"]


def test_app_shows_api_unreachable_when_health_check_fails() -> None:
    import httpx

    with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
        at = AppTest.from_file(APP_PATH).run()

    assert not at.exception
    assert any("unreachable" in e.value for e in at.sidebar.error)


def test_asking_a_question_renders_answer_and_citations() -> None:
    health_response = Mock(status_code=200)
    query_response = Mock(status_code=200)
    query_response.raise_for_status = Mock()
    query_response.json.return_value = {
        "answer": "Incoterms define buyer/seller responsibilities.",
        "citations": [
            {
                "doc_name": "Incoterms Guide",
                "page_number": 1,
                "score": 0.9,
                "text_snippet": "snippet text",
            }
        ],
        "needs_knowledge_search": True,
        "needs_tracking_lookup": False,
        "tracking_info": None,
    }

    with patch("httpx.get", return_value=health_response), patch(
        "httpx.post", return_value=query_response
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("what are the incoterms?").run()

    assert not at.exception
    assert len(at.chat_message) == 2
    user_message, assistant_message = at.chat_message
    assert user_message.markdown[0].value == "what are the incoterms?"
    assert "Incoterms define buyer/seller responsibilities." in assistant_message.markdown[0].value


def test_asking_a_question_renders_tracking_info() -> None:
    health_response = Mock(status_code=200)
    query_response = Mock(status_code=200)
    query_response.raise_for_status = Mock()
    query_response.json.return_value = {
        "answer": "Your package is in transit.",
        "citations": [],
        "needs_knowledge_search": False,
        "needs_tracking_lookup": True,
        "tracking_info": {"status": "In Transit", "origin": "Bonn, DE"},
    }

    with patch("httpx.get", return_value=health_response), patch(
        "httpx.post", return_value=query_response
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("where is my package?").run()

    assert not at.exception
    assert len(at.chat_message) == 2


def test_answer_with_query_id_shows_feedback_buttons() -> None:
    health_response = Mock(status_code=200)
    query_response = Mock(status_code=200)
    query_response.raise_for_status = Mock()
    query_response.json.return_value = {
        "query_id": "q1",
        "answer": "Incoterms define buyer/seller responsibilities.",
        "citations": [],
        "needs_knowledge_search": True,
        "needs_tracking_lookup": False,
        "tracking_info": None,
    }

    with patch("httpx.get", return_value=health_response), patch(
        "httpx.post", return_value=query_response
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("what are the incoterms?").run()

    assert not at.exception
    assert len(at.button) == 2
    assert {b.label for b in at.button} == {"👍", "👎"}


def test_clicking_thumbs_up_submits_feedback_and_hides_buttons() -> None:
    health_response = Mock(status_code=200)
    query_response = Mock(status_code=200)
    query_response.raise_for_status = Mock()
    query_response.json.return_value = {
        "query_id": "q1",
        "answer": "Incoterms define buyer/seller responsibilities.",
        "citations": [],
        "needs_knowledge_search": True,
        "needs_tracking_lookup": False,
        "tracking_info": None,
    }
    feedback_response = Mock(status_code=204)
    feedback_response.raise_for_status = Mock()

    with patch("httpx.get", return_value=health_response), patch(
        "httpx.post", side_effect=[query_response, feedback_response]
    ) as mock_post:
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("what are the incoterms?").run()
        up_button = next(b for b in at.button if b.label == "👍")
        at = up_button.click().run()

    assert not at.exception
    mock_post.assert_any_call(
        "http://localhost:8000/feedback", json={"query_id": "q1", "vote": "up"}, timeout=5.0
    )
    assert len(at.button) == 0
    assert any("Thanks for the feedback" in c.value for c in at.caption)
