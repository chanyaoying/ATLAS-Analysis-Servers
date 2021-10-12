function main() {
    var myConnector = tableau.makeConnector();

    myConnector.getSchema = function (schemaCallback) {
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
        
        schemaCallback([testSchema]);
    };

    myConnector.getData = function (table, doneCallback) {
        $.getJSON("http://localhost:5000/analyse/test/AAPL,FB,AMZN", function (resp) {
            const tableData = resp.data;
            table.appendRows(tableData);
            doneCallback();
        });
    };

    tableau.registerConnector(myConnector);

    $(document).ready(function () {
        $("#ATLASsubmitButton").click(function () {
            tableau.connectionName = "ATLAS Analysis Feed";
            tableau.submit();
        });
    });

};

main()