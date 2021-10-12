const express = require('express')
const cors = require('cors')
const app = express()
const port = 5000

const analysisController = require("./controller/analysisController")

app.use(cors())

app.get("/testpoint", (req, res) => {
    res.send("Connection established")
})

app.use("/analyse", analysisController)


app.listen(port, () => {
    console.log(`ATLAS Analysis server listening at http://localhost:${port}`)
})