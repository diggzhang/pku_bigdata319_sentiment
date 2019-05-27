test:
	python sentiment_comments_predict.py
	head *.txt
	wc -l *.txt
