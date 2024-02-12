# To build Jars
- Prerequisite: Java >= 1.8 is needed. Please make sure the JAVA_HOME environment path has been set.
- First, install IoTDB locally:
```
git clone -b research/M4-visualization http://github.com/apache/iotdb.git
mvn clean install -DskipTests -pl -distribution
```

- Then, set the `finalName` and `mainClass` in the pom.xml as `WriteData`/`QueryData`.

- Next, run `mvn clean package`, and then `WriteData-jar-with-dependencies.jar`/`QueryData-jar-with-dependencies.jar` will be ready in the target folder generated.

- Finally, rename them as `WriteData-0.12.4.jar`/`QueryData-0.12.4.jar` respectively, and copy them to the "jars" folder.
