package plan.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TripInfo {

    private String startingPlace;

    private String endPlace;

    private Date departureTime;

    public TripInfo(){
        //Default Constructor
    }

}
