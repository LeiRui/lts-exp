package org.apache.iotdb.jarCode;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class ExtractUCRData {

  public static void main(String[] args) throws IOException {
    String inPath = args[0];
    System.out.println(inPath);

    String outPath = args[1];
    System.out.println(outPath);

    File f = new File(inPath);
    String line = null;
    BufferedReader reader = new BufferedReader(new FileReader(f));
    PrintWriter writer = new PrintWriter(outPath);
    long globalTimestamp = 0;
    while ((line = reader.readLine()) != null) {
      String[] splitStr = line.split("\\s+");
      for (int i = 1; i < splitStr.length; i++) { // 从1开始，不要第0个位置那是分类标识
        globalTimestamp++;
        double double_value = Double.parseDouble(splitStr[i]);
        writer.println(globalTimestamp + "," + double_value);
      }
    }
    reader.close();
    writer.close();
  }

}
