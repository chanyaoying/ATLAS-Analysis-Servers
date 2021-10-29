function main() {
    var myConnector = tableau.makeConnector();

    myConnector.getSchema = function (schemaCallback) {

        const inputs = JSON.parse(tableau.connectionData)
        const tickers = inputs.tickers;

        // Test table
        const test_cols = [{
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
            id: "portfolioTest",
            alias: "Portfolio Allocation (initial)",
            columns: test_cols
        };

        // Auto ARIMA
        const arima_cols = [{
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "date",
                dataType: tableau.dataTypeEnum.date
            },
            {
                id: "predictionPrice",
                alias: "Price",
                dataType: tableau.dataTypeEnum.float
            },
            // {
            //     id: "predicted",
            //     alias: "Predicted?",
            //     dataType: tableau.dataTypeEnum.integer
            // }
        ]
        const arimaSchema = {
            id: "arimaPrediction",
            alias: `Auto ARIMA for ${tickers.join()}`,
            columns: arima_cols
        }

        // Company Info
        const companyInfo_cols = [{
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "company_name",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "sector",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "website",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "summary",
                dataType: tableau.dataTypeEnum.string
            },
        ]
        const companyInfoSchema = {
            id: "companyInfo",
            alias: `Company info of ${tickers.join()}`,
            columns: companyInfo_cols
        }

        // Financial Statements
        const financialStatements_cols = [{
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "type",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "period",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "breakdown",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "value",
                dataType: tableau.dataTypeEnum.float
            },
        ]
        const financialStatementsSchema = {
            id: "financialStatements",
            alias: `Financial Statements of ${tickers.join()}`,
            columns: financialStatements_cols
        }

        // Markowitz Portfolio Theory
        const markowitzPortfolioTheory_cols = [{
                id: "ticker",
                dataType: tableau.dataTypeEnum.string
            },
            {
                id: "weight",
                dataType: tableau.dataTypeEnum.float
            },
            {
                id: "annualised_return",
                dataType: tableau.dataTypeEnum.float
            },
            {
                id: "annualised_volatility",
                dataType: tableau.dataTypeEnum.float
            },
            {
                id: "type",
                dataType: tableau.dataTypeEnum.string
            }
        ]
        const markowitzPortfolioTheorySchema = {
            id: "markowitzPortfolioTheory",
            alias: `Portfolio Allocation Table of ${tickers.join()}`,
            columns: markowitzPortfolioTheory_cols,
        }

        // Add schemas
        schemaCallback([testSchema, arimaSchema, companyInfoSchema, financialStatementsSchema, markowitzPortfolioTheorySchema]);
    };

    myConnector.getData = function (table, doneCallback) {
        const inputs = JSON.parse(tableau.connectionData)
        const tickers = inputs.tickers;
        const amounts = inputs.amounts;
        const analyses = inputs.analyses;

        const schemaTranslationTable = {
            "Display Portfolio": "portfolioTest",
            "Price Prediction (Auto ARIMA)": "arimaPrediction",
            "Company Information": "companyInfo",
            "Financial Statements": "financialStatements",
            "Find Optimal Portfolio Allocation": "markowitzPortfolioTheory",
        }

        const activeSchemas = analyses.map(x => schemaTranslationTable[x])

        for (const activeSchema of activeSchemas) {
            if (table.tableInfo.id === activeSchema) {
                const apiCall = `http://localhost:5000/${activeSchema}/${tickers}/${amounts}`;
                $.getJSON(apiCall, function (response) {
                    const tableData = response.data;
                    table.appendRows(tableData);
                    doneCallback();
                })
            }
        }
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