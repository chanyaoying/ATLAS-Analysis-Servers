from flask import Flask
from flask_cors import CORS

port = 5001

app = Flask(__name__)
CORS(app)


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/<string:tickers>/<string:amounts>", methods=['GET'])
def home(tickers, amounts):
    print(tickers, amounts)
    tickers_list = tickers.split(',')
    amounts_list = list(map(float, amounts.split(',')))
    summed = sum(amounts_list)
    weights_list = list(map(lambda n: n/summed, amounts_list))

    return {
            "meta": {
                "table_name": f"Test table of {tickers}",
            },
            "data": [{"ticker": tickers_list[i], "weight": weights_list[i], "amount": amounts_list[i]} for i in range(len(tickers_list))],
        }, 200


if __name__ == "__main__":
    app.run(host="localhost", port=port, debug=True)
