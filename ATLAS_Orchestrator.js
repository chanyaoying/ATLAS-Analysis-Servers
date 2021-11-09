const express = require('express')
const cors = require('cors')
const axios = require('axios')
const app = express()
const port = 5000

const config = require('./config')
const hostname = config.hostname


function reorderJSONKeys(object) {
    const data = object.data
    const cols = object.meta.columns
    const newData = []
    for (let row of data) {
        const newRow = {}
        for (let col of cols) {
            newRow[col] = row[col]
        }
        newData.push(newRow)
    }
    return {meta: object.meta, data: newData}
}


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
        res.send(reorderJSONKeys(response.data))
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
        res.send(reorderJSONKeys(response.data))
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})


app.get("/companyInfo/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_companyInfo:${config.ports['companyInfo']}/${tickers}`
    // const apiCall = `http://${hostname}:${config.ports['companyInfo']}/${tickers}`
    try {
        const response = await axios.get(apiCall)
        res.send(reorderJSONKeys(response.data))
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})


app.get("/financialStatements/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_financialStatements:${config.ports['financialStatements']}/${tickers}`
    // const apiCall = `http://${hostname}:${config.ports['financialStatements']}/${tickers}`
    try {
        const response = await axios.get(apiCall)
        res.send(reorderJSONKeys(response.data))
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})


app.get("/markowitzPortfolioTheory/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_markowitzPortfolioTheory:${config.ports['markowitzPortfolioTheory']}/${tickers}`
    // const apiCall = `http://${hostname}:${config.ports['markowitzPortfolioTheory']}/${tickers}`
    try {
        const response = await axios.get(apiCall)
        res.send(reorderJSONKeys(response.data))
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})

app.get("/portfolioRebalancingBySector/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const apiCall = `http://ATLAS_service_portfolioRebalancingBySector:${config.ports['portfolioRebalancingBySector']}/${tickers}/${amounts}`
    // const apiCall = `http://${hostname}:${config.ports['portfolioRebalancingBySector']}/${tickers}/${amounts}`
    try {
        const response = await axios.get(apiCall)
        res.send(reorderJSONKeys(response.data))
    } catch (error) {
        console.error(error)
        res.status(400)
    }
})


app.listen(port, () => {
    console.log(`ATLAS Analysis server listening at http://${hostname}:${port}`)
})