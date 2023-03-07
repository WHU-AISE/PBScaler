package preserve.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class Ticket {

    private int seatNo;

    private String startStation;

    private String destStation;

    public Ticket(){
        //Default Constructor.
    }

}
