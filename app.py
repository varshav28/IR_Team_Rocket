from flask import Flask, render_template, request, redirect, url_for
from query_final import query_text

app = Flask(__name__)

global x
global res
x = 10
res = []

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    global x
    global res
    search_term = request.form["input"]
    print(search_term)
    res = query_text(search_term)
    res2 = res[:10]  
    return render_template('results.html', res=res2)

@app.route('/search/results/more', methods=['GET', 'POST'])
def more():
    global x
    global res
    res2 = res[:x]
    if request.method == 'POST':
        if request.form.get("loadmore") == "Load More":
            print('loading more...')
            x += 20
            res2 = res[:x]
            return render_template('results.html', res=res2)
    return render_template('results.html', res=res2)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)