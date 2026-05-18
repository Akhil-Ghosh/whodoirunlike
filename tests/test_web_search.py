from whodoirunlike.web_search import extract_youtube_ids_from_html, video_id_from_url


def test_extract_youtube_ids_dedupes_in_order():
    html = """
    <a href="/watch?v=ABCDEFGHIJK">One</a>
    <a href="/watch?v=ABCDEFGHIJK">Duplicate</a>
    <a href="/watch?v=ZYXWVUTSRQP">Two</a>
    """

    assert extract_youtube_ids_from_html(html, limit=10) == ["ABCDEFGHIJK", "ZYXWVUTSRQP"]


def test_video_id_from_url_handles_youtube_watch_url():
    assert video_id_from_url("https://www.youtube.com/watch?v=ABCDEFGHIJK&pp=test") == "ABCDEFGHIJK"

