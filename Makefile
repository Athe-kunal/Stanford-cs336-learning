.PHONY: add
add:
	git add -n . ':!*.jsonl' ':!*.ipynb'
	git add . ':!*.jsonl' ':!*.ipynb'