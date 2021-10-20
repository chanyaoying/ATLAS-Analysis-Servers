const express = require('express')
const cors = require('cors')
const axios = require('axios')
const app = express()
const port = 5000

const config = require('./config')
const hostname = config.hostname

app.use(cors())

app.get("/testpoint", (req, res) => {
    res.send("Connection established")
})

// Routes
app.get("/portfolioTest/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_portfolioDisplay:${config.ports['portfolioTest']}/${tickers}/${amounts}`
    // const apiCall = `http://${hostname}:${config.ports['portfolioTest']}/${tickers}/${amounts}`
    try {
        const response = await axios.get(apiCall)
        res.send(response.data)
    } catch (error) {
        console.error(error)
        res.status(400)
    };
})


app.get("/arimaPrediction/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_autoARIMA:${config.ports['arimaPrediction']}/${tickers}`
    // const apiCall = `http://${hostname}:${config.ports['arimaPrediction']}/${tickers}`
    try {
        const response = await axios.get(apiCall)
        res.send(response.data)
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})


app.listen(port, () => {
    console.log(`ATLAS Analysis server listening at http://${hostname}:${port}`)
})