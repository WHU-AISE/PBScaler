package travelplan.entity;

import lombok.Data;

import java.util.Date;

/**
 * @author fdse
 */
@Data
public class TripResponse {

    private TripId tripId;

    private String trainTypeId;

    private String startingStation;

    private String terminalStation;

    private Date startingTime;

    private Date endTime;

    private int economyClass;

    private int confortClass;

    private String priceForEconomyClass;

    private String priceForConfortClass;

    public TripResponse(){
        //Default Constructor
    }

}
