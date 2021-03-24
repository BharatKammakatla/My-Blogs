## ✴️ Selenium Java Automation Framework - How to build ?

QA Automation teams use Test Automation Frameworks for their day-to-day test execution activities. There are several types of automation frameworks and technologies in the market currently and they are continuously evolving. This blog intends to explain the process of building an Automation Framework in java which includes some of the most commonly used tools like selenium, testNG, maven, page object model, extent repots, etc. I would be listing the steps in an ordered manner, however you can download the entire working framework code from my  [GitHub Repo link](https://github.com/BharatKammakatla/SeleniumJavaFramework.git).


-  Create Maven Project.
-  Add dependencies.
-  Create Base Files: base class, global config properties file, browser drivers, base test.
-  Implement Page Object Model: create Page Object classes to store objects of each page separately.
-  Parameterise test data with an external data source(ex: excel sheet, csv, database, etc.)
-  Implement validations using TestNG assertions.
-  Create a testng.xml file: create a suite containing all the tests.
-  Add initialization(`@BeforeTest`) and teardown(`@AfterTest`) methods.
-  Integrate testng.xml into maven pom.xml: On doing so, whenever we trigger maven file(mvn test), it will trigger testng.xml file and that will inturn trigger all the test cases.
-  Add logs using log4j.
-  Modify testng.xml to create individual tests instead of all tests wrapped into a single test tag.
- Tweak framework to support parallel execution. Initialize a local webdriver instance to support parallel execution. Add `parallel='tests'` attribure in testng.xml to run tests parallely.
-  Add screenshots on failures using TestNG Listeners.
-  Add TestNG listener information to testng.xml file.
-  Add extent reports to TestNG listener.
-  Making f/w and extent reports thread safe using `ThreadLocal` class.

📚 Framework code: Refer to my  [GitHub repository](https://github.com/BharatKammakatla/SeleniumJavaFramework.git)

💻  GitHub Profile: [https://github.com/BharatKammakatla](https://github.com/BharatKammakatla)

👨‍💻 Website :  [https://bharatkammakatla.com](https://bharatkammakatla.com) 