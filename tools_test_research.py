#!/usr/bin/env python3
from mad.web_research import run_web_research, ResearchConfig

def main():
    block, sources, artifacts = run_web_research(
        "Should the death penalty be abolished? provide evidence on deterrence and wrongful convictions",
        ResearchConfig(searxng_url="http://127.0.0.1:8080", search_top_k=5, fetch_top_k=2),
    )
    print(block)
    print(f"\nSources count: {len(sources)}")
    print(f"Fetched pages: {len(artifacts.get('fetched_pages', {}))}")

if __name__ == "__main__":
    main()
