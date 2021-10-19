module.exports = {
    hostname: "localhost",
    services: {
        "Test": "test",
        "Price Prediction (Auto ARIMA)": "arima"
    },
    ports: {
        "test": 5001,
        "arima": 5002,
    }
}