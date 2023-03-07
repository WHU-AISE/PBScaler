package foodsearch.entity;


import lombok.Data;

import java.io.Serializable;

@Data
public class Food implements Serializable{

    public Food(){
        //Default Constructor
    }

    private String foodName;
    private double price;

}
