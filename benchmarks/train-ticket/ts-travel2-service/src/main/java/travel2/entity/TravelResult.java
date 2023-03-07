package travel2.entity;

import java.util.Map;
import lombok.*;

/**
 * @author fdse
 */
@Data
public class TravelResult {

    private boolean status;

    private double percent;

    private TrainType trainType;

    private Map<String,String> prices;

    public TravelResult(){
        //Default Constructor
    }

    public boolean isStatus() {
        return status;
    }

}
