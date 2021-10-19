function main() {
    var myConnector = tableau.makeConnector();

    myConnector.getSchema = function (schemaCallback) {

        // Test table
        const test_cols = [{
            id: "date",
            dataType: tableau.dataTypeEnum.string
        }, {
            id: "price",
            alias: "price data",
            dataType: tableau.dataTypeEnum.float
        }];
        const testSchema =  {
            id: "testFeed",
            alias: "Test Table",
            columns: test_cols
        };
        
        // Add schemas
        schemaCallback([testSchema]);
    };

    myConnector.getData = function (table, doneCallback) {
        const inputs = JSON.parse(tableau.connectionData)
        const tickers = inputs.tickers;
        const amounts = inputs.amounts;
        const analyses = inputs.analyses;
        const apiCall = `http://localhost:5000/analyse/test/${tickers}`;

        $.getJSON(apiCall, function (resp) {
            const tableData = resp.data;
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