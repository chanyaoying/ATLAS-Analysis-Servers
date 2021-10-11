(function () {
    var myConnector = tableau.makeConnector();

    myConnector.getSchema = function (schemaCallback) {
        var cols = [{
            id: "date",
            dataType: tableau.dataTypeEnum.string
        }, {
            id: "price",
            alias: "price data",
            dataType: tableau.dataTypeEnum.float
        }];

        var tableSchema = {
            id: "testFeed",
            alias: "Just for testing.",
            columns: cols
        };

        schemaCallback([tableSchema]);
    };

    myConnector.getData = function (table, doneCallback) {
        $.getJSON("http://127.0.0.1:5000/test", function (resp) {

            const tableData = resp.data;

            table.appendRows(tableData);
            doneCallback();
        });
    };

    tableau.registerConnector(myConnector);

    $(document).ready(function () {
        $("#ATLASsubmitButton").click(function () {
            tableau.connectionName = "ATLAS Main Feed";
            tableau.submit();
        });
    });

})();