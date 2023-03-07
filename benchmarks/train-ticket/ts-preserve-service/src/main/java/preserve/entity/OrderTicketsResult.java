package preserve.entity;

import lombok.Data;

/**
 * @author fdse
 */
@Data
public class OrderTicketsResult {

    private boolean status;

    private String message;

    private Order order;

    public OrderTicketsResult(){
        //Default Constructor
    }

}
