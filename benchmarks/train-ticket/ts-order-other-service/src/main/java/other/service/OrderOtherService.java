package other.service;

import edu.fudan.common.util.Response;
import org.springframework.http.HttpHeaders;
import other.entity.*;

import java.util.Date;
import java.util.UUID;

/**
 * @author fdse
 */
public interface OrderOtherService {

    Response findOrderById(UUID id, HttpHeaders headers);

    Response create(Order newOrder, HttpHeaders headers);

    Response updateOrder(Order order, HttpHeaders headers);

    Response saveChanges(Order order, HttpHeaders headers);

    Response cancelOrder(UUID accountId, UUID orderId, HttpHeaders headers);

    Response addNewOrder(Order order, HttpHeaders headers);

    Response deleteOrder(String orderId, HttpHeaders headers);

    Response getOrderById(String orderId, HttpHeaders headers);

    Response payOrder(String orderId, HttpHeaders headers);

    Response getOrderPrice(String orderId, HttpHeaders headers);

    Response modifyOrder(String orderId, int status, HttpHeaders headers);

    Response getAllOrders(HttpHeaders headers);

    Response getSoldTickets(Seat seatRequest, HttpHeaders headers);

    Response queryOrders(QueryInfo qi, String accountId, HttpHeaders headers);

    Response queryOrdersForRefresh(QueryInfo qi, String accountId, HttpHeaders headers);

    Response alterOrder(OrderAlterInfo oai, HttpHeaders headers);

    Response queryAlreadySoldOrders(Date travelDate, String trainNumber, HttpHeaders headers);

    Response checkSecurityAboutOrder(Date checkDate, String accountId, HttpHeaders headers);

    void initOrder(Order order, HttpHeaders headers);
}
