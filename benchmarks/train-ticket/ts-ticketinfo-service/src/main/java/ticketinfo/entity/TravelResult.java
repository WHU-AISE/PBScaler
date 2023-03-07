package ticketinfo.entity;

import java.util.Map;
import lombok.*;

@Data
public class TravelResult {

    private boolean status;

    private double percent;

    private TrainType trainType;

    private Map<String,String> prices;

    private String message;

    public TravelResult(){
        //Default Constructor
    }

}
