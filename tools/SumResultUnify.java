package org.apache.iotdb.tools;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

public class SumResultUnify {

  public static void main(String[] args) throws IOException {
    String mac = args[0]; // sumResultMAC.csv
    String cpv = args[1]; // sumResultCPV.csv
    String out = args[2];

    BufferedReader macReader = new BufferedReader(new FileReader(mac));
    BufferedReader cpvReader = new BufferedReader(new FileReader(cpv));
    PrintWriter printWriter = new PrintWriter(new FileWriter(out));
    String macLine;
    String cpvLine;
    String appendLine;
    boolean isHeader = true;
    while ((macLine = macReader.readLine()) != null) {
      cpvLine = cpvReader.readLine();

      if (isHeader) {
        macLine = "MAC_" + macLine;
        cpvLine = "CPV_" + cpvLine;
        isHeader = false;
      }

      appendLine = macLine + "," + cpvLine;
      printWriter.println(appendLine);
    }
    printWriter.close();
    macReader.close();
    cpvReader.close();
  }
}
