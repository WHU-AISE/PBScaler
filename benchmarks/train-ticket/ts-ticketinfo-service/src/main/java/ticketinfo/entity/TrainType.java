package ticketinfo.entity;

import lombok.Data;

@Data
public class TrainType {

    private String id;

    private int economyClass;
    private int confortClass;

    private int averageSpeed;

    public TrainType(){

    }

    public TrainType(String id, int economyClass, int confortClass) {
        this.id = id;
        this.economyClass = economyClass;
        this.confortClass = confortClass;
    }

}
