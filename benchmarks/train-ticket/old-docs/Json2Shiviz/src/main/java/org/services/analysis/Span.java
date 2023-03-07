package org.services.analysis;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by Administrator on 2017/7/22.
 */
public class Span {
    private String traceId;
    private String spanId;
    private String parentId;
    private String spanname;
    private List<String> childs;
    private List<HashMap<String,String>> logs;

    public Span(String traceId, String spanId, String parentId, String spanname) {
        this.traceId = traceId;
        this.spanId = spanId;
        this.parentId = parentId;
        this.spanname = spanname;
        logs = new ArrayList<HashMap<String,String>>();
    }

    public void addLog(HashMap<String,String> log){
        logs.add(log);
    }

    public List<HashMap<String,String>> getLogs(){
        return logs;
    }

    public String getTraceId() {
        return traceId;
    }

    public String getSpanId() {
        return spanId;
    }

    public String getParentId() {
        return parentId;
    }

    public List<String> getChilds() {
        return childs;
    }

    public void setChilds(List<String> childs) {
        this.childs = childs;
    }

    public void addChild(String childId){
        childs.add(childId);
    }

    public String getSpanname() {
        return spanname;
    }

    public void setSpanname(String spanname) {
        this.spanname = spanname;
    }
}
