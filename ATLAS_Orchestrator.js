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
app.get("/:analyses/:tickers/:amounts", async (req, res) => {
    const params = req.params

    const analyses = params.analyses.split(',') // array
    const tickers = params.tickers // str
    const amounts = params.amounts // str

    const data = analyses.map(async analysisType => {
        const serviceID = config.services[analysisType]
        const apiCall = `http://${hostname}:${config.ports[serviceID]}/${tickers}/${amounts}`
        console.log('apiCall :>> ', apiCall);
        try {
            const response = await axios.get(apiCall)
            return {[analysisType]: response.data}
        } catch (error) {
            // console.error(error)
        }
    });
    Promise.all(data).then(analysisOutputs => {
        let output = {}
        for (let analysisOutput of analysisOutputs) {
            Object.assign(output, analysisOutput)
        }
        console.log('output :>> ', output);
        res.send(output)
    })
})


app.listen(port, () => {
    console.log(`ATLAS Analysis server listening at http://${hostname}:${port}`)
})