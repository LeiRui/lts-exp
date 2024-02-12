package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.iotdb.rpc.IoTDBConnectionException;
import org.apache.iotdb.rpc.StatementExecutionException;
import org.apache.iotdb.session.Session;
import org.apache.iotdb.tsfile.file.metadata.enums.TSDataType;
import org.apache.iotdb.tsfile.file.metadata.enums.TSEncoding;
import org.apache.iotdb.tsfile.write.record.Tablet;
import org.apache.iotdb.tsfile.write.schema.MeasurementSchema;

public class WriteDataUCR {

  /**
   * Before writing data, make sure check the server parameter configurations.
   */
  // Usage: java -jar WriteDataUCR.jar device measurement timestamp_precision dataType valueEncoding iotdb_chunk_point_size filePath
  public static void main(String[] args)
      throws IoTDBConnectionException, StatementExecutionException, IOException {
    String device = args[0];
    System.out.println("[WriteData] device=" + device);

    String measurement = args[1];
    System.out.println("[WriteData] measurement=" + measurement);

    String timestamp_precision = args[2]; // ns, us, ms
    System.out.println("[WriteData] timestamp_precision=" + timestamp_precision);
    if (!timestamp_precision.toLowerCase().equals("ns") && !timestamp_precision.toLowerCase()
        .equals("us") && !timestamp_precision.toLowerCase().equals("ms")) {
      throw new IOException("timestamp_precision only accepts ns,us,ms.");
    }

    String dataType = args[3]; // double only
    System.out.println("[WriteData] dataType=" + dataType);
    TSDataType tsDataType;
    if (dataType.toLowerCase().equals("double")) {
      tsDataType = TSDataType.DOUBLE;
    } else {
      throw new IOException("Data type only accepts double right now.");
    }

    // value encoder
    String valueEncoding = args[4]; // RLE, GORILLA, PLAIN
    System.out.println("[WriteData] valueEncoding=" + valueEncoding);

    int iotdb_chunk_point_size = Integer.parseInt(args[5]);
    System.out.println("[WriteData] iotdb_chunk_point_size=" + iotdb_chunk_point_size);

    // 数据源
    String filePath = args[6];
    System.out.println("[WriteData] filePath=" + filePath);

    //"CREATE TIMESERIES root.vehicle.d0.s0 WITH DATATYPE=INT32, ENCODING=RLE"
    String createSql = String.format("CREATE TIMESERIES %s.%s WITH DATATYPE=%s, ENCODING=%s",
        device,
        measurement,
        tsDataType,
        valueEncoding
    );

    Session session = new Session("127.0.0.1", 6667, "root", "root");
    session.open(false);
    session.executeNonQueryStatement(createSql);

    // this is to make all following inserts unseq chunks
    if (timestamp_precision.toLowerCase().equals("ns")) {
      session.insertRecord(
          device,
          1683616109697000000L, // ns
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
    } else if (timestamp_precision.toLowerCase().equals("us")) {
      session.insertRecord(
          device,
          1683616109697000L, // us
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
    } else { // ms
      session.insertRecord(
          device,
          1683616109697L, // ms
          // NOTE UPDATE TIME DATATYPE! [[update]]. DONT USE System.nanoTime()!
          Collections.singletonList(measurement),
          Collections.singletonList(tsDataType), // NOTE UPDATE VALUE DATATYPE!
          parseValue("0", tsDataType)); // NOTE UPDATE VALUE DATATYPE!
    }
    session.executeNonQueryStatement("flush");

    List<MeasurementSchema> schemaList = new ArrayList<>();
    schemaList.add(
        new MeasurementSchema(measurement, tsDataType, TSEncoding.valueOf(valueEncoding)));
    Tablet tablet = new Tablet(device, schemaList, iotdb_chunk_point_size);
    long[] timestamps = tablet.timestamps;
    Object[] values = tablet.values;

    File f = new File(filePath);
    String line = null;
    BufferedReader reader = new BufferedReader(new FileReader(f));// assume no header
    long globalTimestamp = 0;
    while ((line = reader.readLine()) != null) {
      String[] split = line.split(",");
      globalTimestamp++;
      //  change to batch mode, iotdb_chunk_point_size
      int row = tablet.rowSize++;
      timestamps[row] = globalTimestamp;
      double double_value = Double.parseDouble(split[1]); // get value from real data
      double[] double_sensor = (double[]) values[0];
      double_sensor[row] = double_value;
      if (tablet.rowSize == tablet.getMaxRowNumber()) { // chunk point size
        session.insertTablet(tablet, false);
        tablet.reset();
      }

//      String[] splitStr = line.split("\\s+");
//      for (int i = 1; i < splitStr.length; i++) { // 从1开始，不要第0个位置那是分类标识
//        int row = tablet.rowSize++;
//        globalTimestamp++;
//        timestamps[row] = globalTimestamp;
//        double double_value = Double.parseDouble(splitStr[i]);
//        double[] double_sensor = (double[]) values[0];
//        double_sensor[row] = double_value;
//        if (tablet.rowSize == tablet.getMaxRowNumber()) { // chunk point size
//          session.insertTablet(tablet, false);
//          tablet.reset();
//        }
//      }
    }
    // flush the last Tablet
    if (tablet.rowSize != 0) {
      session.insertTablet(tablet, false);
      tablet.reset();
    }
    session.executeNonQueryStatement("flush");
    session.close();
  }

  public static Object parseValue(String value, TSDataType tsDataType) throws IOException {
    if (tsDataType == TSDataType.DOUBLE) {
      return Double.parseDouble(value);
    } else {
      throw new IOException("data type wrong");
    }
  }
}
