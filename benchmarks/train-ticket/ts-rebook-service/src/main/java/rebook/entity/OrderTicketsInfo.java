package rebook.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class OrderTicketsInfo {

    private String contactsId;

    private String tripId;

    private int seatType;

    private String loginToken;

    private String accountId;

    private Date date;

    private String from;

    private String to;

    public OrderTicketsInfo(){
        //Default Constructor
    }

}
