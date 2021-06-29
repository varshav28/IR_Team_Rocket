from flask import Flask, render_template, request
from query_final import query_text

app = Flask(__name__)
#es = Elasticsearch('10.0.1.10', port=9200)

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    print(search_term)
    res = query_text(search_term)
    print(res)
    return render_template('results.html', res=res[:10] )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)