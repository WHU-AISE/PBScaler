package fdse.microservice.entity;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
public class Travel {

    private Trip trip;
    private String startingPlace;
    private String endPlace;
    private Date departureTime;

    public Travel(){
        //Default Constructor
    }

}
