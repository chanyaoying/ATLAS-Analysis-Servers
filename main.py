import json
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/testpoint")
def test_point():
    return "Connection established.", 200


@app.route("/<string:req>", methods=['GET'])
def home(req):
    if req == "test":
        return json.dumps(
            {
                "meta": {
                    "table_name": "test_analysis_table",
                    "column_names": ['date', 'price']
                },
                "data": [
                    {
                        "date": "2010-05-21",
                        "price": 500.0
                    },
                    {
                        "date": "2010-05-22",
                        "price": 1000.0
                    }
                ]
            }
        ), 200
    else:
        return "Data does not exist.", 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
