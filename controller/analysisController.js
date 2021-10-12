var express = require('express');
var router = express.Router();


// Test
router.get("/test/:tickers", (req, res) => {
    const params = req.params
    const tickers = params.tickers.split(',')
    const data = {
        "meta": {
            "table_name": `Test table of ${tickers.join(", ")}`,
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
            },
            {
                "date": "2010-05-22",
                "price": 1000.0
            },
            {
                "date": "2010-05-21",
                "price": 500.0
            }
        ]
    }
    res.send(data)
})

// Get Ticker Info

module.exports = router;