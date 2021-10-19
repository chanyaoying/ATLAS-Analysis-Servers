function main() {
    var myConnector = tableau.makeConnector();

    myConnector.getSchema = function (schemaCallback) {

        const inputs = JSON.parse(tableau.connectionData)
        const tickers = inputs.tickers;

        // Test table
        const test_cols = [
            {
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "weight",
                alias: "Weight",
                dataType: tableau.dataTypeEnum.float
            },
            {
                id: "amount",
                alias: "Amount",
                dataType: tableau.dataTypeEnum.float
            }
        ];
        const testSchema = {
            id: "testFeed",
            alias: "Initial portfolio allocation (test)",
            columns: test_cols
        };

        // Auto ARIMA
        const arima_cols = [
            {
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "date",
                dataType: tableau.dataTypeEnum.date
            },
            {
                id: "price",
                alias: "Price",
                dataType: tableau.dataTypeEnum.float
            },
            {
                id: "predicted",
                alias: "Predicted?",
                dataType: tableau.dataTypeEnum.integer
            }
        ]
        const arimaSchema = {
            id: "arimaFeed",
            alias: `Auto ARIMA for ${tickers.join()}`,
            columns: arima_cols
        }

        // Add schemas
        schemaCallback([testSchema]);
    };

    myConnector.getData = function (table, doneCallback) {
        const inputs = JSON.parse(tableau.connectionData)
        const tickers = inputs.tickers;
        const amounts = inputs.amounts;
        const analyses = inputs.analyses;
        const apiCall = `http://localhost:5000/${analyses}/${tickers}/${amounts}`;

        $.getJSON(apiCall, function (resp) {
            const tableData = resp['Test'].data; //TODO: use extra tables
            console.log('tableData :>> ', tableData);
            table.appendRows(tableData);
            doneCallback();
        });
    };

    tableau.registerConnector(myConnector);

    $(document).ready(function () {

        $("#ATLASsubmitButton").click(function () {
            const inputs = JSON.parse($("#allInputs").text())

            const connectionName = `${inputs.analyses.join()} of ${inputs.tickers.join()}`;
            tableau.connectionData = JSON.stringify(inputs);
            tableau.connectionName = connectionName;
            tableau.submit();
        });
    });

};

main()