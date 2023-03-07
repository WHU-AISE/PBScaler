package preserveOther.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class Travel {

    private Trip trip;

    private String startingPlace;

    private String endPlace;

    private Date departureTime;

    public Travel(){
        //Default Constructor
    }

}
