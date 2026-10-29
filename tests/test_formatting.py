"""Тесты форматирования ответов для Telegram."""

from __future__ import annotations

from bot.formatting import escape, render_summary, split_html, strip_markdown


def test_escape():
    assert escape("<b>&</b>") == "&lt;b&gt;&amp;&lt;/b&gt;"
    assert escape(None) == ""


def test_split_html_short_text():
    assert split_html("привет") == ["привет"]
    assert split_html("") == [""]


def test_split_html_keeps_tags_balanced():
    text = "<b>" + " ".join(["слово"] * 1200) + "</b>"
    chunks = split_html(text, limit=1000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.count("<b>") == chunk.count("</b>")
        visible = len(chunk.replace("<b>", "").replace("</b>", ""))
        assert visible <= 1000


def test_split_html_reopens_formatting():
    text = "<i>раз два три четыре пять шесть</i>"
    chunks = split_html(text, limit=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("<i>")
        assert chunk.endswith("</i>")


def test_split_html_does_not_break_entities():
    text = "&amp; " * 40
    chunks = split_html(text, limit=12)
    assert len(chunks) > 1
    for chunk in chunks:
        # каждый «&» должен начинать целую сущность
        assert "&" not in chunk.replace("&amp;", "")


def test_strip_markdown():
    raw = "**Жирный** и `код`\n## Заголовок\n- пункт\n> цитата"
    cleaned = strip_markdown(raw)
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "##" not in cleaned
    assert "• пункт" in cleaned
    assert "> цитата" not in cleaned


def test_render_summary_escapes_and_contains_sections():
    payload = {
        "summary": "Обсудили <script>",
        "discussion_type": "dispute",
        "key_points": ["тезис & факт"],
        "disputes": [
            {
                "topic": "Сроки",
                "positions": [{"author": "Иван", "position": "за"}],
                "assessment": "риск",
                "recommended_resolution": "перенести",
            }
        ],
        "decisions": [
            {
                "decision": "перенести релиз",
                "rationale": "тесты",
                "owner": "Пётр & Ко",
                "deadline": "вторник",
            }
        ],
        "action_items": [{"task": "дописать тесты", "owner": "Иван", "deadline": "пн"}],
        "open_questions": ["кто дежурит?"],
        "risks": ["срыв демо"],
        "assistant_opinion": "перенос оправдан",
    }
    html = render_summary(
        payload,
        chat_title="Команда <test>",
        period="01.01 — 02.01",
        message_count=5,
        participant_names=["Иван", "Пётр"],
    )

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "Команда &lt;test&gt;" in html
    assert "Споры и позиции" in html
    assert "перенести релиз" in html
    assert "Пётр &amp; Ко" in html
    assert "&amp;amp;" not in html
    assert "Мнение ассистента" in html
    assert html.count("<b>") == html.count("</b>")
