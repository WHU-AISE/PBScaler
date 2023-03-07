package org.services.analysis;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Test {
    private static String SRC_DIR = "D:\\workspace\\microservice\\faults\\F1\\step2_fault_origin_traces";

    public static void  main(String[] args){
        String mircroServiceName = "{traceId=1b114102c6306ef8, spanId=1b114102c6306ef8, hostName=ts-order-service, destName=null, host=ts-order-service, clock={\"ts-order-service\":4,\"ts-sso-service\":2}, dest=null, event=ts-order-service.OrderController.queryOrders, type=ss, parentId=, timestamp=1516255241135800, }";
        Pattern patternNode = Pattern.compile("host=(\\S*),");
        Matcher matcherCase = patternNode.matcher(mircroServiceName);
        if(matcherCase.find()){
            System.out.println(matcherCase.group(1));
        }
    }

    //Get all the test case directory
    private static ArrayList<File> getTestCaseDirList(String path){
        ArrayList<File> testDirs = new ArrayList<File>();
        File rootDir = new File(path);
        File[] dirList = rootDir.listFiles();
        for (int i = 0; i < dirList.length; i++) {
            File file = dirList[i];
            if(file.isDirectory())
                testDirs.add(file);
        }
        return testDirs;
    }
}
