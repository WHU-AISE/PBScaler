package preserveOther.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
@AllArgsConstructor
@Builder
public class OrderTicketsInfo {
    private String accountId;

    private String contactsId;

    private String tripId;

    private int seatType;

    private Date date;

    private String from;

    private String to;

    private int assurance;


    private int foodType = 0;

    private String stationName;

    private String storeName;

    private String foodName;

    private double foodPrice;


    private String handleDate;

    private String consigneeName;

    private String consigneePhone;

    private double consigneeWeight;

    private boolean isWithin;


    public OrderTicketsInfo(){
        //Default Constructor
    }

}
