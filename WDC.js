const express = require('express')
const serveStatic = require('serve-static')
const app = express()
const port = 8888

app.use(serveStatic('WDC', {
    'index': ['connector.html']
}))


app.listen(port, () => {
    console.log(`WDC listening at http://localhost:${port}`)
})