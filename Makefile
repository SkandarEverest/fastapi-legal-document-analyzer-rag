.PHONY: install run mcp clean

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

mcp:
	uv run python -m app.mcp_server

clean:
	rm -rf chroma uploads *.db *.db-journal
